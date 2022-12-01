# Önceden Eğitilmiş Ağlar ve Öğrenme Aktarımı

CNN'leri eğitmek çok zaman alabilir ve bu görev için çok fazla veri gerekir. Ancak zamanın çoğu, bir ağın imgelerden örüntüleri çıkarmak için kullanabileceği en iyi düşük seviyeli filtreleri öğrenmekle harcanır. Doğal bir soru ortaya çıkıyor - bir veri kümesi üzerinde eğitilmiş bir sinir ağını kullanabilir ve tam bir eğitim süreci gerektirmeden farklı imgeleri sınıflandırmak için uyarlayabilir miyiz?

## [Ders öncesi sınavı](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/108)

Bu yaklaşıma **öğrenme aktarımı** adı verilir, çünkü bazı bilgileri bir sinir ağı modelinden diğerine aktarırız. Öğrenme aktarımında, genellikle **ImageNet** gibi bazı büyük imge veri kümelerinde önceden eğitilmiş bir modelle başlarız. Bu modeller, genel imgelerden farklı öznitelikleri ayıklamak için zaten iyi bir iş çıkarabiliyor ve çoğu durumda bu çıkarılan özniteliklerin üzerine bir sınıflandırıcı oluşturmak iyi bir sonuç verebilir.

> ✅ Öğrenme aktarımı, eğitim gibi diğer akademik alanlarda bulabileceğiniz bir terimdir. Bir alandan bilgi alıp diğerine uygulama işlemini ifade eder.

## Öznitelik Çıkarıcı Olarak Önceden Eğitilmiş Modeller

Bir önceki bölümde bahsettiğimiz evrişimli ağlar, her birinin düşük seviyeli piksel kombinasyonlarından (yatay/dikey çizgi veya kontur gibi) başlayarak bir alev gözü gibi şeylere karşılık gelen daha yüksek seviyeli öznitelik kombinasyonlarına kadar imgeden bazı öznitelikleri çıkarması gereken bir dizi katman içerir. CNN'i yeterince büyük genel ve türlü imgeler veri kümesi üzerinde eğitirsek, ağ bu ortak öznitelikleri çıkarmayı öğrenebilmelidir.

Hem Keras hem de PyTorch, çoğu ImageNet imgeleri üzerinde eğitilmiş bazı yaygın mimariler için önceden eğitilmiş sinir ağı ağırlıklarını kolayca yükleyen işlevler içerir. En sık kullanılanlar önceki dersten [CNN Mimarileri](../../07-ConvNets/translations/CNN_Architectures.tr.md) sayfasında açıklanmıştır. Özellikle, aşağıdakilerden birini kullanmayı düşünebilirsiniz:

* **VGG-16/VGG-19** nispeten basit modellerdir ve yine de iyi doğruluk sağlarlar. Genellikle VGG'yi ilk deneme olarak kullanmak, öğrenme aktarımının nasıl çalıştığını görmek için iyi bir seçimdir.
* **ResNet**, Microsoft Research tarafından 2015 yılında önerilen bir model ailesidir. Daha fazla katmana sahiptirler ve bu nedenle daha fazla kaynak kullanırlar.
* **MobileNet**, mobil cihazlar için uygun, küçültülmüş bir model ailesidir. Kaynaklarınız yetersizse ve doğruluktan biraz ödün verebiliyorsanız bunları kullanın.

İşte VGG-16 ağı tarafından bir kedi resminden çıkarılan örnek öznitelikler:

![Features extracted by VGG-16](../images/features.png)

## Kediler ve Köpekler Veri Kümesi

Bu örnekte, gerçek hayattan bir imge sınıflandırma senaryosuna çok yakın olan [Kediler ve Köpekler](https://www.microsoft.com/download/details.aspx?id=54765&WT.mc_id=academic-77998-cacaste) veri kümesini kullanacağız. 

## ✍️ Alıştırma: Öğrenme Aktarımı

İlgili not defterlerinde öğrenme aktarımını iş başında görelim:

* [Öğrenme Aktarımı - PyTorch](TransferLearningPyTorch.tr.ipynb)
* [Öğrenme Aktarımı - TensorFlow](TransferLearningTF.tr.ipynb)

## Karşıt Kediyi Görselleştirme

Önceden eğitilmiş sinir ağı *beyninin* içinde **ideal kedi** kavramları (ayrıca ideal köpek, ideal zebra vb.) dahil olmak üzere farklı modeller içerir. Bir şekilde **bu imgeyi görselleştirmek** ilginç olurdu. Ancak bu basit değildir, çünkü örüntüler tüm ağ ağırlıklarına dağılmıştır ve aynı zamanda hiyerarşik bir yapıda organize edilmiştir.

Deneyebileceğimiz bir yaklaşım, rastgele bir imgeyle başlamak ve ardından **gradyan inişi optimizasyonu** tekniğini kullanarak bu imgeyi, ağın bunun bir kedi olduğunu düşünmeye başlamasını sağlayacak şekilde ayarlamaktır.

![İmge Eniyileme Döngüsü](../images/ideal-cat-loop.png)

Ancak bunu yaparsak rastgele bir gürültüye çok benzer bir şey alırız. Bunun nedeni, görsel olarak bir anlam ifade etmeyenler de dahil olmak üzere, *ağın girdi imgesinin bir kedi olduğunu düşünmesini sağlamanın birçok yolu* olmasıdır. Bu imgeler, bir kedi için tipik olan pek çok örüntü içeriyor olsa da, görsel olarak ayırt edici olmaları için onları baskılayacak hiçbir şey yoktur.

Sonucu iyileştirmek için kayıp fonksiyonuna **varyasyon kaybı** adı verilen başka bir terim ekleyebiliriz. İmgenin komşu piksellerinin ne kadar benzer olduğunu gösteren bir ölçüttür. Varyasyon kaybının en aza indirilmesi, imgeyi daha pürüzsüz hale getirir ve gürültüyü ortadan kaldırır - böylece görsel olarak daha çekici örüntüler ortaya çıkar. İşte yüksek olasılıkla kedi ve zebra olarak sınıflandırılan bu tür "ideal" imgelere birer örnek:

![İdeal Kedi](../images/ideal-cat.png) | ![İdeal Zebra](../images/ideal-zebra.png)
-----|-----
 *İdeal Kedi* | *İdeal Zebra*

Benzer bir yaklaşım, bir sinir ağında sözde **düşmanca saldırılar** gerçekleştirmek için kullanılabilir. Bir sinir ağını kandırmak ve bir köpeği kedi gibi göstermek istediğimizi varsayalım. Bir ağ tarafından bir köpek olarak tanınan köpeğin imgesini alırsak, ağ onu bir kedi olarak sınıflandırmaya başlayana kadar gradyan inişi optimizasyonunu kullanarak onu biraz değiştirebiliriz:

![Köpek resmi](../images/original-dog.png) | ![Kedi olarak sınıflandırılan bir köpeğin resmi](../images/adversarial-dog.png)
-----|-----
*Bir köpeğin orijinal resmi* | *Kedi olarak sınıflandırılan bir köpeğin resmi*

See the code to reproduce the results above in the following notebook:

* [İdeal ve Karşıt Kedi - TensorFlow](AdversarialCat_TF.tr.ipynb)

## Vargılar

Öğrenme aktarımını kullanarak, özelleştirilmiş bir nesne sınıflandırma görevi için bir sınıflandırıcıyı hızlı bir şekilde bir araya getirebilir ve yüksek doğruluk elde edebilirsiniz. Şu anda çözdüğümüz daha karmaşık görevlerin daha yüksek hesaplama gücü gerektirdiğini ve CPU üzerinde kolayca çözülemeyeceğini görebilirsiniz. Bir sonraki ünitede, aynı modeli daha düşük bilgi işlem kaynakları kullanarak eğitmek için daha hafif bir uygulama kullanmayı deneyeceğiz, bu da sadece biraz daha düşük doğrulukla sonuçlanacak.

## 🚀 Kendini Sınama

Ekteki not defterlerinde, bilgi aktarımının benzer eğitim verileriyle (belki yeni bir hayvan türü) en iyi şekilde nasıl çalıştığına dair notlar var. Bilgi aktarım modellerinizin ne kadar iyi veya kötü performans gösterdiğini görmek için tamamen yeni imge türleri ile biraz deneme yapın.

## [Ders sonrası sınavı](https://red-field-0a6ddfd03.1.azurestaticapps.net/quiz/208)

## Gözden Geçirme & Bireysel Çalışma

Modellerinizi eğitmenin başka bir yolu hakkında bilginizi derinleştirmek için [TrainingTricks.tr.md](TrainingTricks.tr.md) sayfasını okuyun.

## [Ödev](../../lab/translations/README.tr.md)

Bu laboratuvarda, 35 cins kedi ve köpek içeren [Oxford-IIIT](https://www.robots.ox.ac.uk/~vgg/data/pets/) gerçek hayat evcil hayvan veri kümesini kullanacağız ve bir öğrenme aktarma sınıflandırıcısı inşa edeceğiz.